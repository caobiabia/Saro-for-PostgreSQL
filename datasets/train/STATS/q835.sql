select  count(*) from comments as c,          postLinks as pl,          posts as p,  		users as u,  		badges as b   where p.Id = pl.RelatedPostId 	and p.Id = c.PostId 	and u.Id = b.UserId 	and u.Id = p.OwnerUserId  AND b.Date>='2010-08-19 02:44:21'::timestamp  AND c.CreationDate<='2014-09-13 13:21:33'::timestamp  AND pl.LinkTypeId=3  AND pl.CreationDate>='2011-04-09 14:27:41'::timestamp  AND p.CommentCount>=0  AND p.CommentCount<=15  AND p.FavoriteCount>=0  AND u.DownVotes=0  AND u.CreationDate<='2014-07-17 10:29:29'::timestamp;
