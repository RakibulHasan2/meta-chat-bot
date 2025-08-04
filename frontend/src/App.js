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
import { Bot, MessageSquare, Settings, Activity, Users, Zap, Brain, Target, ShoppingCart, Package, TrendingUp, Database } from 'lucide-react';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

function App() {
  const [comments, setComments] = useState([]);
  const [pages, setPages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [demoComment, setDemoComment] = useState('');
  const [selectedPageType, setSelectedPageType] = useState('demo_electronics_page');
  const [demoSetupDone, setDemoSetupDone] = useState(false);
  const [paraphraseText, setParaphraseText] = useState('');
  const [paraphrases, setParaphrases] = useState([]);

  // Predefined demo comments for different scenarios
  const demoComments = {
    'demo_electronics_page': [
      "What is the price of iPhone 15 Pro?",
      "Do you have MacBook Air in stock?",
      "What are your business hours?",
      "I need noise canceling headphones",
      "Do you offer warranty on laptops?"
    ],
    'demo_restaurant_page': [
      "How much is the Margherita pizza?",
      "Do you have vegan options?",
      "What time do you close?",
      "I want to make a reservation for 6 people",
      "Do you deliver food?"
    ]
  };

  useEffect(() => {
    loadComments();
    loadPages();
    setupDemoData();
  }, []);

  const setupDemoData = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${BACKEND_URL}/api/demo/setup`, {
        method: 'POST'
      });
      
      if (response.ok) {
        setDemoSetupDone(true);
        setTimeout(() => {
          loadPages();
        }, 1000);
      }
    } catch (error) {
      console.error('Error setting up demo data:', error);
    }
    setLoading(false);
  };

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

  const handleDemoComment = async (customComment = null) => {
    const commentText = customComment || demoComment;
    if (!commentText.trim()) return;
    
    setLoading(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/demo/comment`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          comment_text: commentText,
          page_id: selectedPageType 
        })
      });
      
      if (response.ok) {
        if (!customComment) {
          setDemoComment('');
        }
        setTimeout(() => loadComments(), 1000);
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

  const getIntentColor = (intent) => {
    switch (intent?.toLowerCase()) {
      case 'price_inquiry': return 'bg-green-100 text-green-800 border-green-200';
      case 'product_info': return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'availability': return 'bg-purple-100 text-purple-800 border-purple-200';
      case 'complaint': return 'bg-red-100 text-red-800 border-red-200';
      case 'compliment': return 'bg-pink-100 text-pink-800 border-pink-200';
      case 'hours': return 'bg-orange-100 text-orange-800 border-orange-200';
      case 'location': return 'bg-indigo-100 text-indigo-800 border-indigo-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getSentimentColor = (sentiment) => {
    switch (sentiment?.toLowerCase()) {
      case 'positive': return 'bg-green-100 text-green-800 border-green-200';
      case 'negative': return 'bg-red-100 text-red-800 border-red-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getPageTypeIcon = (pageType) => {
    switch (pageType) {
      case 'demo_electronics_page': return <ShoppingCart className="h-4 w-4" />;
      case 'demo_restaurant_page': return <Package className="h-4 w-4" />;
      default: return <MessageSquare className="h-4 w-4" />;
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
                <Brain className="h-8 w-8 text-indigo-600" />
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-gray-900 to-indigo-600 bg-clip-text text-transparent">
                  Context-Aware AI Auto-Reply
                </h1>
                <p className="text-sm text-gray-600">Intelligent, page-specific comment management</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                <Database className="h-3 w-3 mr-1" />
                Context AI Active
              </Badge>
              <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
                <TrendingUp className="h-3 w-3 mr-1" />
                {demoSetupDone ? 'Demo Ready' : 'Setting up...'}
              </Badge>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 space-y-8">
        {/* Enhanced Stats */}
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
                  <p className="text-sm font-medium text-green-600">Context Matches</p>
                  <p className="text-2xl font-bold text-green-900">
                    {comments.filter(c => c.confidence_score > 0.7).length}
                  </p>
                </div>
                <Target className="h-8 w-8 text-green-500" />
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
                  <p className="text-sm font-medium text-orange-600">Avg Confidence</p>
                  <p className="text-2xl font-bold text-orange-900">
                    {comments.length > 0 
                      ? Math.round((comments.reduce((acc, c) => acc + (c.confidence_score || 0), 0) / comments.length) * 100)
                      : 0}%
                  </p>
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
              <span>Context Demo</span>
            </TabsTrigger>
            <TabsTrigger value="knowledge" className="flex items-center space-x-2">
              <Database className="h-4 w-4" />
              <span>Knowledge</span>
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
                  <span>Context-Aware Replies</span>
                </CardTitle>
                <CardDescription>
                  AI responses with page-specific knowledge and product awareness
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {comments.length === 0 ? (
                    <div className="text-center py-12 text-gray-500">
                      <Brain className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                      <p>No comments yet. Try the Context Demo to see intelligent responses!</p>
                    </div>
                  ) : (
                    comments.slice(0, 10).map((comment) => (
                      <div key={comment.id} className="p-4 bg-white/80 rounded-lg border border-gray-100 space-y-3">
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <div className="flex items-center space-x-2 mb-2 flex-wrap">
                              <span className="font-medium text-gray-900">{comment.author_name}</span>
                              
                              {/* Page Type Badge */}
                              <Badge className="bg-indigo-100 text-indigo-800 border-indigo-200 text-xs">
                                {getPageTypeIcon(comment.page_id)}
                                <span className="ml-1">
                                  {comment.page_id.includes('electronics') ? 'Electronics' : 
                                   comment.page_id.includes('restaurant') ? 'Restaurant' : 'General'}
                                </span>
                              </Badge>
                              
                              {/* Intent Badge */}
                              {comment.intent && (
                                <Badge className={`text-xs ${getIntentColor(comment.intent)}`}>
                                  {comment.intent.replace('_', ' ')}
                                </Badge>
                              )}
                              
                              {/* Sentiment Badge */}
                              <Badge className={`text-xs ${getSentimentColor(comment.sentiment)}`}>
                                {comment.sentiment}
                              </Badge>
                              
                              {/* Confidence Score */}
                              {comment.confidence_score && (
                                <Badge className={`text-xs ${comment.confidence_score > 0.7 ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'}`}>
                                  {Math.round(comment.confidence_score * 100)}% confident
                                </Badge>
                              )}
                              
                              {comment.replied && (
                                <Badge className="bg-green-100 text-green-800 border-green-200">
                                  <Bot className="h-3 w-3 mr-1" />
                                  Context Reply
                                </Badge>
                              )}
                            </div>
                            
                            <p className="text-gray-700 mb-2">{comment.comment_text}</p>
                            
                            {comment.reply_text && (
                              <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-3 rounded-md border-l-4 border-indigo-200">
                                <p className="text-sm text-indigo-800">
                                  <Brain className="h-4 w-4 inline mr-1" />
                                  <strong>Context-Aware Reply:</strong> {comment.reply_text}
                                </p>
                              </div>
                            )}
                            
                            {/* Show context matches if available */}
                            {comment.context_match && Object.keys(comment.context_match).length > 0 && (
                              <div className="mt-2 p-2 bg-gray-50 rounded text-xs">
                                <strong>Context Matched:</strong> 
                                {comment.context_match.products && (
                                  <span className="ml-1 text-green-600">
                                    Products ({comment.context_match.products.length})
                                  </span>
                                )}
                                {comment.context_match.faqs && (
                                  <span className="ml-1 text-blue-600">
                                    FAQs ({comment.context_match.faqs.length})
                                  </span>
                                )}
                                {comment.context_match.business && (
                                  <span className="ml-1 text-purple-600">Business Info</span>
                                )}
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

          {/* Enhanced Demo Tab */}
          <TabsContent value="demo" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="bg-white/70 backdrop-blur-sm border-gray-200/50">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Target className="h-5 w-5 text-purple-600" />
                    <span>Context-Aware Demo</span>
                  </CardTitle>
                  <CardDescription>
                    Test different page types with specialized responses
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* Page Type Selector */}
                  <div className="space-y-2">
                    <Label>Select Page Type:</Label>
                    <div className="flex space-x-2">
                      <Button
                        variant={selectedPageType === 'demo_electronics_page' ? 'default' : 'outline'}
                        size="sm"
                        onClick={() => setSelectedPageType('demo_electronics_page')}
                        className="flex items-center space-x-1"
                      >
                        <ShoppingCart className="h-4 w-4" />
                        <span>Electronics Store</span>
                      </Button>
                      <Button
                        variant={selectedPageType === 'demo_restaurant_page' ? 'default' : 'outline'}
                        size="sm"
                        onClick={() => setSelectedPageType('demo_restaurant_page')}
                        className="flex items-center space-x-1"
                      >
                        <Package className="h-4 w-4" />
                        <span>Restaurant</span>
                      </Button>
                    </div>
                  </div>
                  
                  {/* Quick Demo Buttons */}
                  <div className="space-y-2">
                    <Label>Quick Test Comments:</Label>
                    <div className="grid grid-cols-1 gap-2">
                      {demoComments[selectedPageType]?.map((comment, index) => (
                        <Button
                          key={index}
                          variant="ghost"
                          size="sm"
                          className="text-left justify-start h-auto p-2 text-sm bg-gray-50 hover:bg-gray-100"
                          onClick={() => handleDemoComment(comment)}
                          disabled={loading}
                        >
                          "{comment}"
                        </Button>
                      ))}
                    </div>
                  </div>
                  
                  <Separator />
                  
                  {/* Custom Comment */}
                  <div className="space-y-2">
                    <Label>Or Type Custom Comment:</Label>
                    <Textarea
                      placeholder="Ask about products, prices, hours, or anything..."
                      value={demoComment}
                      onChange={(e) => setDemoComment(e.target.value)}
                      className="min-h-20"
                    />
                    <Button 
                      onClick={() => handleDemoComment()}
                      disabled={loading || !demoComment.trim()}
                      className="w-full bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700"
                    >
                      {loading ? (
                        <>
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                          Processing...
                        </>
                      ) : (
                        <>
                          <Brain className="h-4 w-4 mr-2" />
                          Generate Context-Aware Reply
                        </>
                      )}
                    </Button>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-white/70 backdrop-blur-sm border-gray-200/50">
                <CardHeader>
                  <CardTitle>Context-Aware Features</CardTitle>
                  <CardDescription>
                    How the AI understands your business context
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-start space-x-3">
                      <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 font-semibold text-sm">1</div>
                      <div>
                        <h4 className="font-medium">Page Context Detection</h4>
                        <p className="text-sm text-gray-600">AI knows if comment is on electronics vs restaurant page</p>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center text-green-600 font-semibold text-sm">2</div>
                      <div>
                        <h4 className="font-medium">Product Knowledge</h4>
                        <p className="text-sm text-gray-600">Knows prices, specs, availability of your products</p>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <div className="w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center text-purple-600 font-semibold text-sm">3</div>
                      <div>
                        <h4 className="font-medium">Intent Classification</h4>
                        <p className="text-sm text-gray-600">Price inquiry, product info, hours, location detection</p>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <div className="w-8 h-8 bg-orange-100 rounded-full flex items-center justify-center text-orange-600 font-semibold text-sm">4</div>
                      <div>
                        <h4 className="font-medium">Business Information</h4>
                        <p className="text-sm text-gray-600">Hours, location, contact info automatically included</p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            <Alert className="bg-gradient-to-r from-blue-50 to-indigo-50 border-indigo-200">
              <Brain className="h-4 w-4 text-indigo-600" />
              <AlertDescription className="text-indigo-800">
                <strong>Context-Aware AI:</strong> This system understands your business type and provides relevant, intelligent responses based on your products, services, and business information.
              </AlertDescription>
            </Alert>
          </TabsContent>

          {/* Knowledge Tab */}
          <TabsContent value="knowledge" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="bg-white/70 backdrop-blur-sm border-gray-200/50">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Database className="h-5 w-5 text-green-600" />
                    <span>Electronics Store Knowledge</span>
                  </CardTitle>
                  <CardDescription>Products and business information</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="text-sm">
                    <strong>Products:</strong>
                    <ul className="mt-1 space-y-1 text-gray-600">
                      <li>• iPhone 15 Pro - $999.99</li>
                      <li>• MacBook Air M3 - $1,199.99</li>
                      <li>• Sony WH-1000XM5 - $399.99</li>
                    </ul>
                  </div>
                  <div className="text-sm">
                    <strong>Business Hours:</strong>
                    <p className="text-gray-600">Mon-Fri: 9AM-8PM, Sat-Sun: 10AM-6PM</p>
                  </div>
                  <div className="text-sm">
                    <strong>Location:</strong>
                    <p className="text-gray-600">123 Tech Street, Silicon Valley, CA</p>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-white/70 backdrop-blur-sm border-gray-200/50">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Package className="h-5 w-5 text-orange-600" />
                    <span>Restaurant Knowledge</span>
                  </CardTitle>
                  <CardDescription>Menu items and restaurant information</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="text-sm">
                    <strong>Menu Items:</strong>
                    <ul className="mt-1 space-y-1 text-gray-600">
                      <li>• Margherita Pizza - $18.99</li>
                      <li>• Spaghetti Carbonara - $22.99</li>
                      <li>• Tiramisu - $8.99</li>
                    </ul>
                  </div>
                  <div className="text-sm">
                    <strong>Business Hours:</strong>
                    <p className="text-gray-600">Daily: 11AM-11PM</p>
                  </div>
                  <div className="text-sm">
                    <strong>Location:</strong>
                    <p className="text-gray-600">456 Food Avenue, Downtown, NY</p>
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
            <p className="mb-2">🧠 Context-Aware AI Facebook Auto-Reply System</p>
            <p className="text-sm">
              Intelligent responses with page-specific knowledge, product awareness, and business context understanding
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;