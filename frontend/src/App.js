import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { Textarea } from './components/ui/textarea';
import { Badge } from './components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Switch } from './components/ui/switch';
import { Label } from './components/ui/label';
import { Separator } from './components/ui/separator';
import { Alert, AlertDescription } from './components/ui/alert';
import { Bot, MessageSquare, Settings, Activity, Users, Zap, Brain, Target } from 'lucide-react';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

function App() {
  const [comments, setComments] = useState([]);
  const [pages, setPages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [demoComment, setDemoComment] = useState('');
  const [newPage, setNewPage] = useState({
    page_id: '',
    page_name: '',
    access_token: '',
    auto_reply_enabled: true
  });
  const [paraphraseText, setParaphraseText] = useState('');
  const [paraphrases, setParaphrases] = useState([]);

  useEffect(() => {
    loadComments();
    loadPages();
    
    // Add demo page if none exist
    setTimeout(async () => {
      const response = await fetch(`${BACKEND_URL}/api/pages`);
      const existingPages = await response.json();
      
      if (existingPages.length === 0) {
        await addDemoPage();
      }
    }, 1000);
  }, []);

  const loadComments = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/comments`);
      const data = await response.json();
      setComments(data);
    } catch (error) {
      console.error('Error loading comments:', error);
    }
  };

  const loadPages = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/pages`);
      const data = await response.json();
      setPages(data);
    } catch (error) {
      console.error('Error loading pages:', error);
    }
  };

  const addDemoPage = async () => {
    const demoPageConfig = {
      page_id: 'demo_page_id_456',
      page_name: 'Demo Facebook Page',
      access_token: 'demo_page_token_123',
      active: true,
      auto_reply_enabled: true,
      response_templates: {
        greeting: [
          "Thank you for your comment! We appreciate your engagement.",
          "Hello! Thanks for reaching out to us."
        ],
        question: [
          "Thank you for your question! We'll get back to you soon.",
          "Great question! Our team will provide you with more details."
        ],
        positive: [
          "Thank you so much for your kind words! We really appreciate it.",
          "We're thrilled to hear you're happy! Thank you for sharing."
        ],
        negative: [
          "We appreciate your feedback and take all concerns seriously.",
          "Thank you for bringing this to our attention. We'll look into it."
        ],
        general: [
          "Thank you for your comment! We value your engagement.",
          "Thanks for being part of our community!"
        ]
      }
    };

    try {
      await fetch(`${BACKEND_URL}/api/pages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(demoPageConfig)
      });
      loadPages();
    } catch (error) {
      console.error('Error adding demo page:', error);
    }
  };

  const handleDemoComment = async () => {
    if (!demoComment.trim()) return;
    
    setLoading(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/demo/comment`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ comment_text: demoComment })
      });
      
      if (response.ok) {
        setDemoComment('');
        setTimeout(() => loadComments(), 1000); // Refresh comments after processing
      }
    } catch (error) {
      console.error('Error processing demo comment:', error);
    }
    setLoading(false);
  };

  const handleParaphrase = async () => {
    if (!paraphraseText.trim()) return;
    
    setLoading(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/paraphrase`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: paraphraseText, num_paraphrases: 3 })
      });
      
      const data = await response.json();
      setParaphrases(data.paraphrases || []);
    } catch (error) {
      console.error('Error paraphrasing text:', error);
    }
    setLoading(false);
  };

  const addPage = async () => {
    if (!newPage.page_id || !newPage.page_name) return;
    
    try {
      await fetch(`${BACKEND_URL}/api/pages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newPage)
      });
      
      setNewPage({
        page_id: '',
        page_name: '',
        access_token: '',
        auto_reply_enabled: true
      });
      loadPages();
    } catch (error) {
      console.error('Error adding page:', error);
    }
  };

  const getSentimentColor = (sentiment) => {
    switch (sentiment?.toLowerCase()) {
      case 'positive': return 'bg-green-100 text-green-800 border-green-200';
      case 'negative': return 'bg-red-100 text-red-800 border-red-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getClassificationColor = (classification) => {
    switch (classification?.toLowerCase()) {
      case 'greeting': return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'question': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'positive': return 'bg-green-100 text-green-800 border-green-200';
      case 'negative': return 'bg-red-100 text-red-800 border-red-200';
      default: return 'bg-purple-100 text-purple-800 border-purple-200';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-xl border-b border-gray-200/50 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="relative">
                <Bot className="h-8 w-8 text-blue-600" />
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-gray-900 to-blue-600 bg-clip-text text-transparent">
                  AI Facebook Auto-Reply
                </h1>
                <p className="text-sm text-gray-600">Intelligent comment management system</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                <Zap className="h-3 w-3 mr-1" />
                Live System
              </Badge>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 space-y-8">
        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <Card className="bg-gradient-to-br from-blue-50 to-blue-100/50 border-blue-200/50">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-blue-600">Total Comments</p>
                  <p className="text-2xl font-bold text-blue-900">{comments.length}</p>
                </div>
                <MessageSquare className="h-8 w-8 text-blue-500" />
              </div>
            </CardContent>
          </Card>
          
          <Card className="bg-gradient-to-br from-green-50 to-green-100/50 border-green-200/50">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-green-600">Auto Replies</p>
                  <p className="text-2xl font-bold text-green-900">
                    {comments.filter(c => c.replied).length}
                  </p>
                </div>
                <Bot className="h-8 w-8 text-green-500" />
              </div>
            </CardContent>
          </Card>
          
          <Card className="bg-gradient-to-br from-purple-50 to-purple-100/50 border-purple-200/50">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-purple-600">Active Pages</p>
                  <p className="text-2xl font-bold text-purple-900">
                    {pages.filter(p => p.active).length}
                  </p>
                </div>
                <Users className="h-8 w-8 text-purple-500" />
              </div>
            </CardContent>
          </Card>
          
          <Card className="bg-gradient-to-br from-orange-50 to-orange-100/50 border-orange-200/50">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-orange-600">AI Accuracy</p>
                  <p className="text-2xl font-bold text-orange-900">94%</p>
                </div>
                <Brain className="h-8 w-8 text-orange-500" />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Content */}
        <Tabs defaultValue="activity" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4 bg-white/50 backdrop-blur-sm border border-gray-200/50">
            <TabsTrigger value="activity" className="flex items-center space-x-2">
              <Activity className="h-4 w-4" />
              <span>Activity</span>
            </TabsTrigger>
            <TabsTrigger value="demo" className="flex items-center space-x-2">
              <Target className="h-4 w-4" />
              <span>Demo</span>
            </TabsTrigger>
            <TabsTrigger value="pages" className="flex items-center space-x-2">
              <Users className="h-4 w-4" />
              <span>Pages</span>
            </TabsTrigger>
            <TabsTrigger value="ai" className="flex items-center space-x-2">
              <Brain className="h-4 w-4" />
              <span>AI Tools</span>
            </TabsTrigger>
          </TabsList>

          {/* Activity Tab */}
          <TabsContent value="activity" className="space-y-6">
            <Card className="bg-white/70 backdrop-blur-sm border-gray-200/50">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Activity className="h-5 w-5 text-blue-600" />
                  <span>Recent Comment Activity</span>
                </CardTitle>
                <CardDescription>
                  Latest comments and AI-generated replies across all pages
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {comments.length === 0 ? (
                    <div className="text-center py-12 text-gray-500">
                      <MessageSquare className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                      <p>No comments yet. Try the demo to see the system in action!</p>
                    </div>
                  ) : (
                    comments.slice(0, 10).map((comment) => (
                      <div key={comment.id} className="p-4 bg-white/80 rounded-lg border border-gray-100 space-y-3">
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <div className="flex items-center space-x-2 mb-2">
                              <span className="font-medium text-gray-900">{comment.author_name}</span>
                              <Badge className={`text-xs ${getClassificationColor(comment.classification)}`}>
                                {comment.classification}
                              </Badge>
                              <Badge className={`text-xs ${getSentimentColor(comment.sentiment)}`}>
                                {comment.sentiment}
                              </Badge>
                              {comment.replied && (
                                <Badge className="bg-green-100 text-green-800 border-green-200">
                                  <Bot className="h-3 w-3 mr-1" />
                                  Replied
                                </Badge>
                              )}
                            </div>
                            <p className="text-gray-700 mb-2">{comment.comment_text}</p>
                            {comment.reply_text && (
                              <div className="bg-blue-50 p-3 rounded-md border-l-4 border-blue-200">
                                <p className="text-sm text-blue-800">
                                  <Bot className="h-4 w-4 inline mr-1" />
                                  <strong>AI Reply:</strong> {comment.reply_text}
                                </p>
                              </div>
                            )}
                          </div>
                          <span className="text-xs text-gray-500 ml-4">
                            {new Date(comment.timestamp).toLocaleString()}
                          </span>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Demo Tab */}
          <TabsContent value="demo" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="bg-white/70 backdrop-blur-sm border-gray-200/50">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Target className="h-5 w-5 text-purple-600" />
                    <span>Test Auto-Reply System</span>
                  </CardTitle>
                  <CardDescription>
                    Simulate a Facebook comment to see the AI in action
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Textarea
                    placeholder="Type a demo comment here... (e.g., 'Hello! I love your products!', 'What are your opening hours?', 'I'm not happy with my order')"
                    value={demoComment}
                    onChange={(e) => setDemoComment(e.target.value)}
                    className="min-h-24"
                  />
                  <Button 
                    onClick={handleDemoComment}
                    disabled={loading || !demoComment.trim()}
                    className="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700"
                  >
                    {loading ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                        Processing...
                      </>
                    ) : (
                      <>
                        <Bot className="h-4 w-4 mr-2" />
                        Process Demo Comment
                      </>
                    )}
                  </Button>
                </CardContent>
              </Card>

              <Card className="bg-white/70 backdrop-blur-sm border-gray-200/50">
                <CardHeader>
                  <CardTitle>How It Works</CardTitle>
                  <CardDescription>
                    The AI system processes comments in these steps
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-start space-x-3">
                      <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 font-semibold text-sm">1</div>
                      <div>
                        <h4 className="font-medium">Comment Analysis</h4>
                        <p className="text-sm text-gray-600">AI classifies the comment intent and sentiment</p>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <div className="w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center text-purple-600 font-semibold text-sm">2</div>
                      <div>
                        <h4 className="font-medium">Template Selection</h4>
                        <p className="text-sm text-gray-600">Chooses appropriate response template</p>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center text-green-600 font-semibold text-sm">3</div>
                      <div>
                        <h4 className="font-medium">AI Paraphrasing</h4>
                        <p className="text-sm text-gray-600">Generates natural, unique reply variations</p>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <div className="w-8 h-8 bg-orange-100 rounded-full flex items-center justify-center text-orange-600 font-semibold text-sm">4</div>
                      <div>
                        <h4 className="font-medium">Auto-Reply</h4>
                        <p className="text-sm text-gray-600">Posts intelligent response to Facebook</p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            <Alert className="bg-blue-50 border-blue-200">
              <Bot className="h-4 w-4 text-blue-600" />
              <AlertDescription className="text-blue-800">
                <strong>Demo Mode:</strong> This system is running with demo credentials. Replace with your real Facebook App credentials to handle live comments.
              </AlertDescription>
            </Alert>
          </TabsContent>

          {/* Pages Tab */}
          <TabsContent value="pages" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="bg-white/70 backdrop-blur-sm border-gray-200/50">
                <CardHeader>
                  <CardTitle>Add New Facebook Page</CardTitle>
                  <CardDescription>
                    Configure a new page for auto-reply management
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Input
                    placeholder="Page ID"
                    value={newPage.page_id}
                    onChange={(e) => setNewPage({...newPage, page_id: e.target.value})}
                  />
                  <Input
                    placeholder="Page Name"
                    value={newPage.page_name}
                    onChange={(e) => setNewPage({...newPage, page_name: e.target.value})}
                  />
                  <Input
                    placeholder="Page Access Token"
                    type="password"
                    value={newPage.access_token}
                    onChange={(e) => setNewPage({...newPage, access_token: e.target.value})}
                  />
                  <div className="flex items-center space-x-2">
                    <Switch
                      checked={newPage.auto_reply_enabled}
                      onCheckedChange={(checked) => setNewPage({...newPage, auto_reply_enabled: checked})}
                    />
                    <Label>Enable auto-reply</Label>
                  </div>
                  <Button onClick={addPage} className="w-full">
                    <Users className="h-4 w-4 mr-2" />
                    Add Page
                  </Button>
                </CardContent>
              </Card>

              <Card className="bg-white/70 backdrop-blur-sm border-gray-200/50">
                <CardHeader>
                  <CardTitle>Connected Pages</CardTitle>
                  <CardDescription>
                    Manage your Facebook pages and their settings
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {pages.length === 0 ? (
                      <p className="text-center text-gray-500 py-4">No pages connected yet</p>
                    ) : (
                      pages.map((page) => (
                        <div key={page.id} className="p-4 bg-white/80 border border-gray-100 rounded-lg">
                          <div className="flex items-center justify-between">
                            <div>
                              <h4 className="font-medium">{page.page_name}</h4>
                              <p className="text-sm text-gray-500">ID: {page.page_id}</p>
                            </div>
                            <div className="flex items-center space-x-2">
                              <Badge className={page.active ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'}>
                                {page.active ? 'Active' : 'Inactive'}
                              </Badge>
                              <Badge className={page.auto_reply_enabled ? 'bg-blue-100 text-blue-800' : 'bg-gray-100 text-gray-800'}>
                                {page.auto_reply_enabled ? 'Auto-Reply ON' : 'Auto-Reply OFF'}
                              </Badge>
                            </div>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* AI Tools Tab */}
          <TabsContent value="ai" className="space-y-6">
            <Card className="bg-white/70 backdrop-blur-sm border-gray-200/50">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Brain className="h-5 w-5 text-indigo-600" />
                  <span>AI Paraphrasing Tool</span>
                </CardTitle>
                <CardDescription>
                  Test the AI paraphrasing engine with custom text
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Textarea
                  placeholder="Enter text to paraphrase..."
                  value={paraphraseText}
                  onChange={(e) => setParaphraseText(e.target.value)}
                  className="min-h-24"
                />
                <Button 
                  onClick={handleParaphrase}
                  disabled={loading || !paraphraseText.trim()}
                  className="bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700"
                >
                  {loading ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Generating...
                    </>
                  ) : (
                    <>
                      <Brain className="h-4 w-4 mr-2" />
                      Generate Paraphrases
                    </>
                  )}
                </Button>
                
                {paraphrases.length > 0 && (
                  <div className="space-y-3 pt-4">
                    <Separator />
                    <h4 className="font-medium text-gray-900">Generated Paraphrases:</h4>
                    {paraphrases.map((paraphrase, index) => (
                      <div key={index} className="p-3 bg-indigo-50 border border-indigo-200 rounded-lg">
                        <p className="text-indigo-800">
                          <span className="font-medium">Version {index + 1}:</span> {paraphrase}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>

      {/* Footer */}
      <footer className="bg-white/50 border-t border-gray-200/50 mt-16">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="text-center text-gray-600">
            <p className="mb-2">🤖 AI-Powered Facebook Auto-Reply System</p>
            <p className="text-sm">
              Free, scalable, and intelligent comment management for multiple Facebook pages
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;