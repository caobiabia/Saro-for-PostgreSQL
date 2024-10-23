select  count(*) from comments as c,  		posts as p,          postHistory as ph where p.Id = c.PostId 	and p.Id = ph.PostId  AND p.CommentCount>=0;
